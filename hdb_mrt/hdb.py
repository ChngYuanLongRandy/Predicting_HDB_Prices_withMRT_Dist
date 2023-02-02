""" This is a helper file for HDB data from data.gov.sg 
"""
from typing import List
import pandas as pd
import zipfile
from io import BytesIO
import requests
import logging 

def retrieve_record(url_base:str, resource_id:str, limit:int, )-> dict:
    """returns a Dataframe from the base url under the resource id
    up to the limit specified. 

    Parameters
    ----------
    url_base : str
        API endpoint
    
    resource_id : 
    
    returns: pd.DataFrame
        Records of the results
    """
    payload = {
        'resource_id' : resource_id,
        'limit' : limit 
    }
    try: 
        response = requests.get(url_base, params=payload)
        data = response.json()
        data = data['result']['records'] # data only found in records under result

        return data

    except Exception as e:
        logging.error(f'Unable to retrieve records due to {e}')

        return None

def retrieve_records(url_base:str, urls: List[str],limit:int)->pd.DataFrame:
    """Returns the entire list of url as a pandas dataframe.
    If there is a link that is broken, it will return None
    Will also add in an additional column of remaining lease into the 
    years that do not have it.

    Args:
        url_base (str): API endpoint for data.gov.sg
        urls (List[str]): list of url
        limit (int): Limits the amount of records being retrieved 
        in each given year

    Returns:
        pd.DataFrame: Compilation of the entire record
    """
    list_of_results = []
    for url in urls:
        result = retrieve_record(url_base, url, limit)
        try:
            list_of_results.append(result)

        except Exception as e:

            logging.error(f'Unable to retrieve records due to {e}')
            return None


    # 2017 to 2014 records with additional column 'remaining lease'
    df_part1 = pd.concat((
        pd.DataFrame(list_of_results[0]),
        pd.DataFrame(list_of_results[1]),
        pd.DataFrame(list_of_results[2])), axis=0)
    # 2014 and before 
    df_part2 = pd.concat((
        pd.DataFrame(list_of_results[3]),
        pd.DataFrame(list_of_results[4])), axis=0)

    df_part1['remaining_lease'] = 99 - (pd.to_datetime(df_part1['month']).dt.year - df_part1['lease_commence_date'].astype(int) )

    # Rearranging cols so that the concatation will work later
    df = df_part1[['remaining_lease']+ [col for col in df_part1.columns if col != 'remaining_lease']]
    df2 = df_part2[['remaining_lease']+ [col for col in df_part2.columns if col != 'remaining_lease']]

    df_combined = pd.concat((df, df2))

    return df_combined

def download_mrt_excelsheet(url:str)->None:
    """
    Takes the URL and downloads the xlsx file in memory and extract the files
    """
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall()

def extract_mrt_excelsheet(default_file_name:str = "Train Station Codes and Chinese Names.xls")->pd.DataFrame:
    download_mrt_excelsheet()
    return pd.read_excel(default_file_name)

def return_geo_details(address_list:List[str])->List[str]:
    geo_details = []
    ONE_MAP_API_ADDRESS = "https://developers.onemap.sg/commonapi/search"
    for address in address_list:
        payload = {'searchVal': address,
        'returnGeom': 'Y',
        'getAddrDetails': 'Y',
        'pageNum':1}

        res= requests.get(ONE_MAP_API_ADDRESS, params=payload)

        try:
            res = res.json()['results'][0] # get only the first result
            full_add = res['ADDRESS']
            lat = float(res['LATITUDE'])
            long = float(res['LONGITUDE'])
            geo_details.append([full_add, lat, long])
        except:
            geo_details.append(['data unavailable','data unavailable','data unavailable'])
    
    return geo_details


def return_one_address(query:str)->dict:
    ONE_MAP_API_ADDRESS = "https://developers.onemap.sg/commonapi/search"
    payload = {'searchVal': query,
    'returnGeom': 'Y',
    'getAddrDetails': 'Y',
    'pageNum':1}

    res= requests.get(ONE_MAP_API_ADDRESS, params=payload)

    try:
        res = res.json()['results'][0] # get only the first result
        full_add = res['ADDRESS']
        lat = float(res['LATITUDE'])
        long = float(res['LONGITUDE'])
        entry = {'address': full_add, 'lat': lat, 'long': long}
    except Exception as e:
        print(f'No results due to {e}')
    
    return entry