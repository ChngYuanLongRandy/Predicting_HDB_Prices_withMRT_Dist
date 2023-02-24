""" This is a helper file to take information from LTA's datamall
or from onemap api
"""
import pandas as pd
import numpy as np
from typing import List
import socket
import requests
import logging
import zipfile
from io import BytesIO

import haversine.haversine as haversine

from . import hdb

def extract_mrt_excelsheet(mrt_api:str, mrt_excel_file:str)->pd.DataFrame:
    """Reads the zip file of the MRT train names in memory and
    returns a pd Dataframe of the data

    Args:
        mrt_api (str): api endpoint of the lta datamall for train names
        mrt_excel_file (str): name of the excel file

    Returns:
        pd.DataFrame: pd dataframe of the excel file
    """
    try:
        r = requests.get(mrt_api)
        z = zipfile.ZipFile(BytesIO(r.content), 'r')
        # we only need 2 columns
        data = pd.read_excel(z.read(mrt_excel_file)).loc[:,['stn_code', 'mrt_station_english']] 
        return data
    except Exception as e:
        logging.error(f'Unable to extract MRT excelsheet. Error {e}')

def allowed_gai_family():
    """A solution I found online to retrieve results from API much much faster
    The solution : https://discuss.onemap.sg/t/search-api-returning-response-404-after-a-while/1710/4

    Apparently it forces to use IPv4 instead of IPv6
    """
    family = socket.AF_INET    # force IPv4
    return family

def generate_uniqueHDBaddresses(data_gov_api, urls, limit):
    hdb_df = hdb.retrieve_records(data_gov_api, urls, limit)
    # Manual transformation of certain addresses due to search logic
    hdb_df['street_name'] = hdb_df['street_name'].where(hdb_df['street_name']!='MARINE CRES','MARINE CRESCENT VILLE') 

    unique_address = hdb_df['address'].unique()

    return unique_address

def return_closest_mrt_distances(unique_address_geo_pd_valid:pd.DataFrame, stations_pd:pd.DataFrame)->pd.DataFrame:
    res = []
    for address in unique_address_geo_pd_valid.values:
        lat,long = address[2] , address[3]
        mrt = return_closest_mrt_distance((lat,long), stations_pd)
        res.append([mrt['closest_mrt'], mrt['distance']])

    assert len(unique_address_geo_pd_valid) == len(res)
    unique_address_geo_pd_valid.reset_index(drop=True)
    combined_pd = pd.concat((unique_address_geo_pd_valid.reset_index(drop=True),pd.DataFrame(res)), axis=1, ignore_index=True)

    return combined_pd


def return_closest_mrt_distance(coor:List[float], stations:pd.DataFrame)->dict:
    """ with one set of coordinates (lat, long) returns the minimum distance
    to the MRT along with the MRT name
    """
    res = np.zeros((len(stations),1))
    for idx, station in enumerate(stations.values):
        station_coor = (station[3], station[4])
        res[idx] = haversine(coor, (station_coor))

    min_distance_idx = np.argmin(res)
    
    return {'closest_mrt': stations.values[min_distance_idx][0],'distance': round(res[min_distance_idx][0],3)}

def return_mrt_with_geo(one_map_url:str,stations:pd.DataFrame)->pd.DataFrame:
    """ returns the full address along with the lat and long of the MRT station.
    This set of data is effectively the excel spreadsheet with relevant columns only
    and the geo location and full name appended on it.

    Function will attempt to return the correct address by checking against the station
    code associated with the mrt station

    Cols of Dataframe returned:
    station_name, station_code, address, lat, long

    """

    geo_details = {}
    for station in stations.values:
        station_code = station[0].lower()
        station_name = station[1].lower()
        for train in ['mrt station', 'lrt station']:
            payload = {'searchVal': f"{station_name} {train} ({station_code}) ",
            'returnGeom': 'Y',
            'getAddrDetails': 'Y',
            'pageNum':1}

            res= requests.get(one_map_url, params=payload)

            try:
                res = res.json()['results'][0] # get only the first result
                full_add = res['ADDRESS']
                lat = float(res['LATITUDE'])
                long = float(res['LONGITUDE'])
                
                # only keep the address if the search val matches and there is nothing in the result
                if (station_name in res['SEARCHVAL'].lower()) and (station_name not in geo_details): 
                    entry = {'stn_code': station_code, 'address': full_add, 'lat': lat, 'long': long}
                    geo_details[station_name] = entry
                    
            except Exception as e:
                logging.debug(f"Search for {station_name} {train} skipped")
    
    stations_pd = pd.DataFrame(geo_details).T.reset_index().rename(columns={'index':'station_name'})

    return stations_pd




def return_unique_address_geo_mrt_post_pre2000(hdb_df:pd.DataFrame, ):
    """Takes all of the HDB data, 
    """

    # searches for unique address. No point searching twice for the same data
    hdb_df['address'] = hdb_df['block'] + " " + hdb_df['street_name']
    unique_address = hdb_df['address'].unique()

    pass

def return_geo_many_address(one_map_url:str, address_list:List[str])->List[str]:
    """Returns addresses. Calls `return_geo_one_address` for each address and returns
    a List

    Args:
        one_map_url (str): base api endpoint of the one map sg
        address_list (List[str]): a list of address

    Returns:
        List[str]: all of the address with full address, lat and long
    """
    geo_details = []
    for address in address_list:

        res = return_geo_one_address(one_map_url,address)

        geo_details.append([res['address'], res['lat'], res['long']])
    
    return geo_details

def return_geo_one_address(one_map_url:str, query:str)->dict:
    """Return geo details of one address given an address
    It is known to not perform well if additional information is given
    like street.
    Takes only the first result that is given from the API.
    For MRT, it must be <station name> <MRT/LRT Station> (station code)
    for example Ang Mo kio MRT station (NS24)
    otherwise there might be zero or inaccurate result

    Args:
        one_map_url (str): base api endpoint of the one map sg
        query (str): address e.g 325 Ang Mo Kio

    Returns:
        dict: address with lat, long and full address
    """
    payload = {'searchVal': query,
    'returnGeom': 'Y',
    'getAddrDetails': 'Y',
    'pageNum':1}

    res= requests.get(one_map_url, params=payload)

    try:
        res = res.json()['results'][0] # get only the first result
        full_add = res['ADDRESS']
        lat = float(res['LATITUDE'])
        long = float(res['LONGITUDE'])
        address = {'address': full_add, 'lat': lat, 'long': long}
    except Exception as e:
        logging.debug(f'No results for {query} due to message: {e}')
        address = {'address': 'na', 'lat': 'na', 'long': 'na'}
    
    return address